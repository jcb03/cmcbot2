import discord
import asyncio
import logging
from hybrid_chat_bot import SimpleHinglishChatBot

# Discord Bot Token (replace with your actual token)
BOT_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"

# Setup Discord intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Queue for processing messages (no parallel processing)
queue = asyncio.Queue()
bot = None  # Will hold your chatbot instance

# Setup logging
logging.basicConfig(
    filename='discord_bot.log', 
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

async def process_queue():
    """Process messages from queue one at a time"""
    await client.wait_until_ready()
    
    while not client.is_closed():
        try:
            message, content = await queue.get()
            response = ""
            
            print(f"Processing: {content}")
            
            # Handle commands
            if content.lower() in ['quit', 'exit', 'stop']:
                await message.channel.send("👋 Bye! Bot shutting down...")
                await client.close()
                break
                
            elif content.lower() == 'stats':
                response = bot.get_stats()
                
            elif content.lower() == 'reset':
                bot.hybrid_store.force_reset()
                response = "🔄 Reset completed! Please restart the bot to reprocess data."
                
            elif content.lower().startswith('doc '):
                try:
                    doc_id = int(content.split(' ', 1)[1].strip())
                    doc_content = bot.get_document_by_id(doc_id)
                    if doc_content:
                        response = f"📄 **Document {doc_id}:**\n``````"
                    else:
                        response = f"❌ Document {doc_id} not found!"
                except (ValueError, IndexError):
                    response = "❌ Usage: `doc <doc_id>`\nExample: `doc 12345`"
                    
            elif content.lower().startswith('lookup '):
                try:
                    doc_id = int(content.split(' ', 1)[1].strip())
                    doc_info = bot.lookup_document(doc_id)
                    response = doc_info if doc_info else f"❌ Document {doc_id} not found!"
                except (ValueError, IndexError):
                    response = "❌ Usage: `lookup <doc_id>`\nExample: `lookup 12345`"
                    
            else:
                # Regular question
                response = bot.ask_question(content)
            
        except Exception as e:
            response = f"❌ Error processing request: {str(e)}"
            logging.error(f"Error processing '{content}': {e}")
        
        # Send response (split if too long)
        if len(response) > 2000:
            # Discord message limit is 2000 chars
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(response)
        
        queue.task_done()

@client.event
async def on_ready():
    """Initialize bot when Discord client is ready"""
    global bot
    
    print(f"🤖 Logged in as {client.user}")
    print("🔄 Initializing Hybrid Hinglish Chat Bot...")
    
    try:
        # Initialize your chatbot
        bot = SimpleHinglishChatBot("channel_export.json")
        
        if bot.setup():
            print("✅ Bot setup completed!")
            print("💬 Interactive Mode!")
            print("Commands: 'stats', 'reset', 'quit', 'doc <doc_id>', 'lookup <doc_id>'")
            print("📝 Logs visible in: discord_bot.log and hybrid_search.log")
            print("=" * 60)
        else:
            print("❌ Bot setup failed!")
            await client.close()
            
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        await client.close()

@client.event
async def on_message(message):
    """Handle incoming Discord messages"""
    # Ignore messages from the bot itself
    if message.author == client.user:
        return
    
    # Only process Direct Messages (private messages)
    if isinstance(message.channel, discord.DMChannel):
        content = message.content.strip()
        
        # Log the message
        logging.info(f"User {message.author} ({message.author.id}): {content}")
        print(f"📩 Message from {message.author}: {content}")
        
        # Add to queue for processing
        await queue.put((message, content))
        await message.channel.send("🤖 Your message is queued for processing...")

# Start the queue processor
client.loop.create_task(process_queue())

# Run the Discord bot
if __name__ == "__main__":
    print("🚀 Starting Discord Bot...")
    client.run(BOT_TOKEN)
