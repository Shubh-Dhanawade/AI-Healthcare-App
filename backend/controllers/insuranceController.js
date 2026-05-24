const supabase = require('../config/supabase');
const pdfParse = require('pdf-parse');
const { v4: uuidv4 } = require('uuid');

const uploadInsurance = async (req, res) => {
  try {
    if (!req.file || req.file.mimetype !== 'application/pdf') {
      return res.status(400).json({ message: 'Please upload a PDF document' });
    }

    const file = req.file;
    const userId = req.user.id;
    const fileName = `${userId}/${uuidv4()}-${file.originalname}`;

    // 1. Upload to Supabase Storage (Assumes 'insurance' bucket exists)
    const { data: uploadData, error: uploadError } = await supabase
      .storage
      .from('insurance')
      .upload(fileName, file.buffer, {
        contentType: file.mimetype,
      });

    if (uploadError) throw uploadError;

    const { data: { publicUrl } } = supabase
      .storage
      .from('insurance')
      .getPublicUrl(fileName);

    // 2. Extract Text from PDF
    const pdfData = await pdfParse(file.buffer);
    const extractedText = pdfData.text;

    // 3. AI Analysis using Ollama locally
    let aiSummary = "AI analysis not available. Please ensure Ollama is running.";
    let risksAndExclusions = "N/A";
    let coverageDetails = "N/A";
    
    try {
      const response = await fetch(`${process.env.OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llama3',
          prompt: `Analyze the following insurance document. Extract important coverage details, detect any risks or exclusions, and provide a short summary:\n\n${extractedText.substring(0, 4000)}`, // limit text for prompt
          stream: false,
        }),
      });
      const data = await response.json();
      aiSummary = data.response; // In a real scenario, ask for JSON format and parse
    } catch (err) {
      console.log('Ollama not reachable:', err.message);
    }

    // 4. Save to Database
    const { data: insuranceDoc, error: dbError } = await supabase
      .from('insurance_documents')
      .insert([{
        user_id: userId,
        document_url: publicUrl,
        extracted_text: extractedText,
        ai_summary: aiSummary,
      }])
      .select()
      .single();

    if (dbError) throw dbError;

    res.status(201).json(insuranceDoc);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: error.message });
  }
};

const getInsuranceDocs = async (req, res) => {
  try {
    const { data: docs, error } = await supabase
      .from('insurance_documents')
      .select('*')
      .eq('user_id', req.user.id)
      .order('created_at', { ascending: false });

    if (error) throw error;
    res.json(docs);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { uploadInsurance, getInsuranceDocs };
