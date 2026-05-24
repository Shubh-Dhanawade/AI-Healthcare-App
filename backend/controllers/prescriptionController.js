const supabase = require('../config/supabase');
const Tesseract = require('tesseract.js');
const { v4: uuidv4 } = require('uuid');

const uploadPrescription = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'Please upload an image' });
    }

    const file = req.file;
    const userId = req.user.id;
    const fileName = `${userId}/${uuidv4()}-${file.originalname}`;

    // 1. Upload to Supabase Storage (Assumes 'prescriptions' bucket exists)
    const { data: uploadData, error: uploadError } = await supabase
      .storage
      .from('prescriptions')
      .upload(fileName, file.buffer, {
        contentType: file.mimetype,
      });

    if (uploadError) throw uploadError;

    const { data: { publicUrl } } = supabase
      .storage
      .from('prescriptions')
      .getPublicUrl(fileName);

    // 2. OCR using Tesseract
    const { data: { text: extractedText } } = await Tesseract.recognize(
      file.buffer,
      'eng',
      { logger: m => console.log(m) }
    );

    // 3. AI Explanation using Ollama locally (placeholder if Ollama is not actually running, but implementing call)
    let aiSummary = "AI summary not available. Please ensure Ollama is running.";
    try {
      const response = await fetch(`${process.env.OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llama3',
          prompt: `Summarize the following prescription text and explain the medicines simply:\n${extractedText}`,
          stream: false,
        }),
      });
      const data = await response.json();
      aiSummary = data.response;
    } catch (err) {
      console.log('Ollama not reachable:', err.message);
    }

    // 4. Save to Database
    const { data: prescription, error: dbError } = await supabase
      .from('prescriptions')
      .insert([{
        user_id: userId,
        image_url: publicUrl,
        extracted_text: extractedText,
        ai_summary: aiSummary,
      }])
      .select()
      .single();

    if (dbError) throw dbError;

    res.status(201).json(prescription);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: error.message });
  }
};

const getPrescriptions = async (req, res) => {
  try {
    const { data: prescriptions, error } = await supabase
      .from('prescriptions')
      .select('*')
      .eq('user_id', req.user.id)
      .order('created_at', { ascending: false });

    if (error) throw error;
    res.json(prescriptions);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { uploadPrescription, getPrescriptions };
