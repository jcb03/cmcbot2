import discord
import asyncio
from queue import Queue
import os
from pathlib import Path
from collections import defaultdict
import time

# Load environment variables
def load_env():
    env_path = Path('.') / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.strip().split('=', 1)
                        os.environ[key.strip()] = val.strip().strip('"\'')

class HinglishDiscordBot(discord.Client):
    def __init__(self, bot_logic, *args, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(*args, intents=intents, **kwargs)

        self.bot_logic = bot_logic
        self.message_queue = Queue()
        self.processing = False

        # SLOWMODE SETTINGS
        self.user_cooldowns = defaultdict(float)
        self.cooldown_duration = 30  # 30 seconds cooldown
        self.admin_users = set()

    async def on_ready(self):
        print(f'📱 CMC Lore Bot logged in as {self.user}')
        print(f'🎯 Connected to {len(self.guilds)} servers')
        print('💬 Interactive Mode!')
        print("📋 Commands: '.lore stats', '.lore doc <doc_id>', '.lore lookup <doc_id>', '.lore reset'")
        print("🤔 Questions: '.lore who is theabbie?', '.lore competitive programming discuss hua?'")
        print('📝 Logs are available in hybrid_search.log')
        print('=' * 60)

    async def on_message(self, message):
        # Ignore bot's own messages
        if message.author == self.user:
            return

        content = message.content.strip()
        
        # ONLY respond to messages starting with '.lore'
        if not content.lower().startswith('.lore'):
            return  # Ignore all other messages
        
        # CHECK SLOWMODE (Skip for admins)
        user_id = message.author.id
        if user_id not in self.admin_users:  # Admins bypass cooldown
            current_time = time.time()
            last_command_time = self.user_cooldowns[user_id]
            
            if current_time - last_command_time < self.cooldown_duration:
                remaining_time = self.cooldown_duration - (current_time - last_command_time)
                await message.add_reaction('🕒')  # Add clock reaction
                await message.channel.send(
                    f"🕒 **Slowmode Active!** {message.author.mention}, please wait **{remaining_time:.1f} seconds** before using `/lore` again."
                )
                return
            
            # Update user's cooldown
            self.user_cooldowns[user_id] = current_time

        # Extract the command/query after '/lore'
        query = content[5:].strip()  # Remove '.lore' and leading spaces
        
        if not query:
            await message.channel.send("⚠️ **Please provide a command or question after '.lore'**\n\n**Examples:**\n• `.lore stats` - Show bot statistics\n• `.lore who is theabbie?` - Ask questions\n• `.lore doc 12345` - Get document content\n• `.lore lookup 67890` - Lookup document details\n• `.lore reset` - Reset bot data")
            return
        
        # Add to queue with the extracted query
        self.message_queue.put((message.channel, query, message.author))
        
        # Process queue if not already processing
        if not self.processing:
            await self.process_message_queue()

    async def process_message_queue(self):
        """Process messages sequentially"""
        self.processing = True
        
        while not self.message_queue.empty():
            channel, content, author = self.message_queue.get()
            
            print(f'🗣️  {author.name}: .lore {content}')
            
            try:
                response = await self.handle_user_command(content)
                
                # Split long responses for Discord limits
                if len(response) > 2000:
                    for i in range(0, len(response), 1900):
                        chunk = response[i:i+1900]
                        if i > 0:
                            chunk = "...(continued)\n" + chunk
                        await channel.send(chunk)
                else:
                    await channel.send(response)
                    
            except Exception as e:
                error_msg = f"❌ Error processing command: {str(e)}"
                await channel.send(error_msg)
                print(f"Error: {e}")
            
            self.message_queue.task_done()
        
        self.processing = False

    async def handle_user_command(self, content):
        """Handle user commands and queries"""
        cmd = content.lower().strip()
        
        # Handle specific commands
        if cmd == 'stats':
            return f"📊 **CMC Lore Bot Statistics:**\n\n{self.bot_logic.get_stats()}"
            
        elif cmd == 'reset':
            self.bot_logic.hybrid_store.force_reset()
            return '🔄 **Bot Reset Complete!**\nRestart the bot to reprocess data from scratch.'
            
        elif cmd == 'quit':
            await self.close()
            return '👋 **Goodbye!** CMC Lore Bot is shutting down...'
            
        elif cmd == 'help':
            help_text = """ **CMC Lore Bot - Command Guide:**

**📋 Utility Commands:**
• `.lore stats` - Show bot statistics and data info
• `.lore reset` - Reset and reprocess all data
• `.lore help` - Show this help message

**🔍 Document Commands:**
• `.lore doc <doc_id>` - Get document content by ID
• `.lore lookup <doc_id>` - Get detailed document info by ID

**💬 Ask Questions:**
• `.lore who is theabbie?` - Ask about community members
• `.lore competitive programming discuss hua?` - Ask about topics
• `.lore job career advice mili?` - Ask about any discussions

**Examples from your server:**
• `.lore theabbie kon hai?`
• `.lore CMC server mein kaun active hai?`
• `.lore priyansh ke baare mein kya baat hui?`"""
            return help_text
            
        elif cmd.startswith('lookup '):
            try:
                doc_id = int(cmd.split(' ', 1)[1].strip())
                result = self.bot_logic.lookup_document(doc_id)
                return result if result else f'❌ Document ID {doc_id} not found!'
            except (ValueError, IndexError):
                return '❌ **Invalid command!**\nUsage: `.lore lookup <doc_id>`\nExample: `.lore lookup 12345`'
                
        elif cmd.startswith('doc '):
            try:
                doc_id = int(cmd.split(' ', 1)[1].strip())
                doc_content = self.bot_logic.get_document_by_id(doc_id)
                if doc_content:
                    return f'📄 **Document {doc_id}:**\n``````'
                else:
                    return f'❌ Document ID {doc_id} not found!'
            except (ValueError, IndexError):
                return '❌ **Invalid command!**\nUsage: `.lore doc <doc_id>`\nExample: `.lore doc 12345`'
                
        else:
            # Treat as question/query about CMC server lore
            print(f"📝 Processing lore query: {content}")
            response = self.bot_logic.ask_question(content)
            
            # Add lore context to response
            lore_response = f"{response}"
            return lore_response

    async def close(self):
        """Clean shutdown"""
        print('🛑 CMC Lore Bot shutting down...')
        await super().close()

# Bot Logic Adapter 
class DiscordBotAdapter:
    def __init__(self, hinglish_chat_bot):
        self.bot = hinglish_chat_bot
        
    def get_stats(self):
        return self.bot.get_stats()
        
    def lookup_document(self, doc_id):
        return self.bot.lookup_document(doc_id)
        
    def get_document_by_id(self, doc_id):
        content = self.bot.get_document_by_id(doc_id)
        return content if content else None
        
    def ask_question(self, question):
        return self.bot.ask_question(question)

# Main Discord Bot Runner
async def run_discord_bot():
    load_env()
    
    # Import your existing bot
    from hybrid_chat_bot import SimpleHinglishChatBot
    
    # Initialize your existing bot
    print("🚀 Initializing CMC Lore Bot...")
    hinglish_bot = SimpleHinglishChatBot("channel_export.json")
    
    if not hinglish_bot.setup():
        print("❌ Bot setup failed!")
        return
    
    print("✅ CMC Lore Bot setup complete!")
    
    # Create adapter
    bot_logic = DiscordBotAdapter(hinglish_bot)
    
    # Create Discord bot
    discord_bot = HinglishDiscordBot(bot_logic)
    
    # Get Discord token
    discord_token = os.getenv('DISCORD_BOT_TOKEN')
    if not discord_token:
        print("❌ DISCORD_BOT_TOKEN not found in .env file!")
        return
    
    # Run Discord bot
    try:
        await discord_bot.start(discord_token)
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        await discord_bot.close()

if __name__ == "__main__":
    print("🔥 Starting CMC Lore Bot...")
    asyncio.run(run_discord_bot())
