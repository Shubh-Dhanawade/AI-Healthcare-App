const supabase = require('../config/supabase');

const chatWithAI = async (req, res) => {
  const { message } = req.body;
  const userId = req.user.id;

  try {
    // 1. Save User Message
    await supabase.from('chat_history').insert([{
      user_id: userId,
      role: 'user',
      content: message
    }]);

    // 2. Fetch context (e.g., last 5 messages, latest prescription/insurance summaries)
    // For simplicity, we just use the current message here, but you can enhance this
    let aiResponseText = "AI is currently unavailable.";
    
    try {
      const ollamaResponse = await fetch(`${process.env.OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llama3',
          prompt: `You are an AI Healthcare Assistant. The user says: ${message}`,
          stream: false,
        }),
      });
      const data = await ollamaResponse.json();
      aiResponseText = data.response;
    } catch (err) {
      console.log('Ollama not reachable:', err.message);
    }

    // 3. Save AI Response
    const { data: aiMessage, error } = await supabase.from('chat_history').insert([{
      user_id: userId,
      role: 'assistant',
      content: aiResponseText
    }]).select().single();

    if (error) throw error;

    res.json(aiMessage);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: error.message });
  }
};

const getChatHistory = async (req, res) => {
  try {
    const { data: history, error } = await supabase
      .from('chat_history')
      .select('*')
      .eq('user_id', req.user.id)
      .order('created_at', { ascending: true });

    if (error) throw error;
    res.json(history);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { chatWithAI, getChatHistory };
