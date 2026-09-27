"""Read-only agent for the real Integrated India account.

It logs in to the broker's web portal (mobile number, one-time password asked for on Telegram, MPIN), reads what the
portal shows, and logs out. It never places, modifies or cancels orders and never moves money: it only opens an
allow-listed set of read pages, clicks nothing after login, and a network guard aborts any request that looks like a
transaction. It shares no code with the paper-trading bot, and nothing in the paper bot imports it.
"""
