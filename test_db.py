from database.setup import store_chat, get_history
store_chat("User", "Hello, world!")
history = get_history()
print(history)

