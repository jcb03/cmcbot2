from hybrid_chat_bot import SimpleHinglishChatBot

def test_bot_locally():
    print("🧪 Testing bot locally...")
    
    # Initialize bot
    bot = SimpleHinglishChatBot("channel_export.json")
    
    if not bot.setup():
        print("❌ Setup failed!")
        return
    
    # Test commands
    test_commands = [
        "is sana hindu or muslim?"
    ]
    
    print("\n💬 Testing Commands:")
    print("=" * 50)
    
    for cmd in test_commands:
        print(f"\n🗣️  Test: {cmd}")
        
        if cmd == "stats":
            response = bot.get_stats()
        elif cmd.startswith("lookup"):
            doc_id = int(cmd.split()[1])
            response = bot.lookup_document(doc_id)
        elif cmd.startswith("doc"):
            doc_id = int(cmd.split()[1])
            response = bot.get_document_by_id(doc_id)
        else:
            response = bot.ask_question(cmd)
        
        print(f"🤖 Response: {response[:2000]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_bot_locally()
