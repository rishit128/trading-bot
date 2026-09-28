#!/usr/bin/env bash
# Provisions one EC2 instance for the trading bot + real-portfolio agent, and deploys the current checkout to it.
# Idempotent: re-running reuses whatever already exists (by name/tag) instead of creating duplicates.
#
# Needs: AWS CLI configured with credentials that can create IAM roles/policies, security groups and EC2 instances,
# plus ec2-instance-connect:SendSSHPublicKey (used only to recover access if deploy/*.pem is ever lost) — an admin or
# PowerUser-ish IAM user/role — run `aws configure` (or `aws sso login`) yourself first; this script never asks for or
# sees your AWS keys. Also needs `ssh`, `scp`, `tar`, `ssh-keygen` (all present in Git Bash on Windows).
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-ap-south-1}"                 # Mumbai: closest to NSE/the Integrated portal
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.small}"          # 2 GB RAM: comfortable for Chromium + the paper bot together
NAME="ai-trading-bot"
KEY_FILE="deploy/${NAME}-key.pem"                   # gitignored; the only copy of the SSH private key
SECRET_NAME="ai-trading-bot/integrated-portfolio"

aws() { command aws --region "$REGION" --output json "$@"; }

echo "== AWS identity =="
aws sts get-caller-identity --query "[Account,Arn]" --output text

echo "== IAM: role + instance profile (Secrets Manager access to one named secret only) =="
if ! aws iam get-role --role-name "$NAME" >/dev/null 2>&1; then
    aws iam create-role --role-name "$NAME" --assume-role-policy-document '{
        "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
        >/dev/null
fi
aws iam put-role-policy --role-name "$NAME" --policy-name portfolio-secret \
    --policy-document file://deploy/ec2-secrets-policy.json >/dev/null
if ! aws iam get-instance-profile --instance-profile-name "$NAME" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$NAME" --role-name "$NAME" >/dev/null
    sleep 10  # IAM is eventually consistent; a brand-new instance profile is not always usable immediately
fi

echo "== Network: default VPC/subnet, a security group with SSH from your current IP only =="
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --query "Subnets[0].SubnetId" --output text)
MY_IP=$(curl -s https://checkip.amazonaws.com)/32
SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$NAME" Name=vpc-id,Values="$VPC_ID" \
    --query "SecurityGroups[0].GroupId" --output text)
if [ "$SG_ID" = "None" ]; then
    SG_ID=$(aws ec2 create-security-group --group-name "$NAME" --vpc-id "$VPC_ID" \
        --description "ai-trading-bot: SSH from the operators IP only; the bot itself only makes outbound calls" \
        --query GroupId --output text)
fi
# Replace any previous SSH rule with one for the current IP (idempotent: ignore "already exists"/"not found").
aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP" >/dev/null 2>&1 || true

echo "== SSH key pair (local, never registered with AWS as an EC2 key-pair object) =="
# AWS EC2 "key pairs" only take effect at first boot, baked into the AMI's cloud-init via --key-name. Recreating the
# AWS-side object later (the old behaviour here) does nothing for an instance already running with the old one, so a
# lost $KEY_FILE permanently locked you out even though the instance was fine. Instead, authorized_keys is seeded
# directly: via user-data on first boot below, or via EC2 Instance Connect further down on a later run. Losing
# $KEY_FILE is then always recoverable without losing the instance or its data.
if [ ! -f "$KEY_FILE" ]; then
    ssh-keygen -t ed25519 -f "$KEY_FILE" -N "" -q -C "$NAME"
fi
if [ ! -f "${KEY_FILE}.pub" ]; then
    # A private key from before this script generated its own (e.g. one AWS previously handed back) has no .pub
    # alongside it; derive one rather than assume ssh-keygen always created this file.
    chmod 600 "$KEY_FILE"
    ssh-keygen -y -f "$KEY_FILE" > "${KEY_FILE}.pub"
fi
PUBKEY=$(cat "${KEY_FILE}.pub")

echo "== EC2 instance (Ubuntu 22.04, Docker installed via user-data) =="
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters Name=tag:Name,Values="$NAME" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query "Reservations[0].Instances[0].InstanceId" --output text)
INSTANCE_IS_NEW=false
if [ "$INSTANCE_ID" = "None" ]; then
    INSTANCE_IS_NEW=true
    AMI_ID=$(aws ec2 describe-images --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
        --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text)
    INSTANCE_ID=$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
        --security-group-ids "$SG_ID" --subnet-id "$SUBNET_ID" \
        --iam-instance-profile "Name=$NAME" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
        --user-data "#!/bin/bash
set -e
mkdir -p /home/ubuntu/.ssh
echo '$PUBKEY' >> /home/ubuntu/.ssh/authorized_keys
chown -R ubuntu:ubuntu /home/ubuntu/.ssh
chmod 700 /home/ubuntu/.ssh
chmod 600 /home/ubuntu/.ssh/authorized_keys
apt-get update -y
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable\" \
    > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
usermod -aG docker ubuntu" \
        --query "Instances[0].InstanceId" --output text)
elif [ "$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "Reservations[0].Instances[0].State.Name" --output text)" = "stopped" ]; then
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
fi
echo "waiting for the instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

echo "== Elastic IP (a fixed public address, associated with this instance) =="
EIP_ALLOC=$(aws ec2 describe-addresses --filters Name=tag:Name,Values="$NAME" --query "Addresses[0].AllocationId" --output text)
if [ "$EIP_ALLOC" = "None" ]; then
    EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME}]" \
        --query AllocationId --output text)
fi
aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$EIP_ALLOC" >/dev/null
IP=$(aws ec2 describe-addresses --allocation-ids "$EIP_ALLOC" --query "Addresses[0].PublicIp" --output text)
echo "instance: $INSTANCE_ID   static IP: $IP"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@$IP"

if [ "$INSTANCE_IS_NEW" = false ] && ! $SSH -o BatchMode=yes "true" >/dev/null 2>&1; then
    echo "== $KEY_FILE isn't trusted by the existing instance yet (a new local key was just generated, or this is a"
    echo "   different machine) -- registering it via EC2 Instance Connect, a one-time temporary key push =="
    AZ=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "Reservations[0].Instances[0].Placement.AvailabilityZone" --output text)
    IC_KEY=$(mktemp -u)  # -u: a name only, not a file -- ssh-keygen would ask to overwrite mktemp's own placeholder
    ssh-keygen -t ed25519 -f "$IC_KEY" -N "" -q
    aws ec2-instance-connect send-ssh-public-key --instance-id "$INSTANCE_ID" --instance-os-user ubuntu \
        --ssh-public-key "file://${IC_KEY}.pub" --availability-zone "$AZ" >/dev/null
    ssh -i "$IC_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@"$IP" \
        "mkdir -p ~/.ssh && grep -qxF '$PUBKEY' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys" \
        || { echo "Instance Connect push failed -- check that your IAM user/role has ec2-instance-connect:SendSSHPublicKey"; \
             echo "and that the security group still allows SSH from this machine's IP."; rm -f "$IC_KEY" "${IC_KEY}.pub"; exit 1; }
    rm -f "$IC_KEY" "${IC_KEY}.pub"
    echo "   $KEY_FILE is now trusted by the instance."
fi

echo "== Waiting for SSH and Docker (cloud-init can take a couple of minutes on a fresh instance) =="
for _ in $(seq 1 40); do
    if $SSH "command -v docker" >/dev/null 2>&1; then break; fi
    sleep 15
done
$SSH "command -v docker" >/dev/null || { echo "Docker never came up on the instance; check /var/log/cloud-init-output.log over SSH."; exit 1; }

echo "== Packaging this checkout (excluding venv, .git, caches, databases, local secrets) and uploading it =="
TARBALL=$(mktemp).tar.gz
tar --exclude-vcs --exclude=venv --exclude=__pycache__ --exclude='*.pyc' --exclude='*.db' \
    --exclude=backtest_cache.json --exclude=research_cache --exclude=portfolio_data --exclude=nse_files \
    --exclude=logs --exclude=htmlcov --exclude=.pytest_cache --exclude=deploy \
    -czf "$TARBALL" .
$SSH "mkdir -p ~/$NAME"
scp -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new "$TARBALL" "ubuntu@$IP:~/${NAME}.tar.gz"
$SSH "tar -xzf ~/${NAME}.tar.gz -C ~/$NAME && rm ~/${NAME}.tar.gz"
rm -f "$TARBALL"

if ! $SSH "test -f ~/$NAME/.env"; then
    echo "!! No .env on the instance yet. Copy your OPENROUTER_API_KEY / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID there:"
    echo "     scp -i $KEY_FILE .env ubuntu@$IP:~/$NAME/.env"
    echo "   (never the Integrated mobile/MPIN — those go into Secrets Manager, next step below). Then re-run this script."
    exit 1
fi
# boto3 (the portfolio agent's AWS Secrets Manager backend) needs a region; the EC2 instance role gives it credentials
# but not a region automatically. Keep .env matched to wherever this script actually deployed, rather than a value
# baked into the image that could go stale if a future run targets a different AWS_REGION.
$SSH "grep -q '^AWS_DEFAULT_REGION=' ~/$NAME/.env || echo 'AWS_DEFAULT_REGION=$REGION' >> ~/$NAME/.env"

echo "== Building and starting the containers (this rebuilds Chromium + Playwright; a few minutes the first time) =="
$SSH "cd ~/$NAME && sudo docker compose -f docker-compose.aws.yml up -d --build"

cat <<EOF

Deployed. Static IP: $IP   SSH: $SSH

One-time, if not already done: save the Integrated login (mobile/MPIN/customer ID) to AWS Secrets Manager by running,
on the instance:
    $SSH
    cd $NAME && sudo docker compose -f docker-compose.aws.yml exec trading-bot python -m src.portfolio setup

Then send /portfolio to the bot on Telegram as usual. Logs:
    $SSH "cd $NAME && sudo docker compose -f docker-compose.aws.yml logs -f"
EOF
