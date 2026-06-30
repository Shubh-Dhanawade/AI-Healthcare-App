const supabase = require('../config/supabase');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

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

    // 2. OCR using PaddleOCR (via python CLI wrapper)
    const tempDir = path.join(__dirname, '../uploads');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    const tempFilePath = path.join(tempDir, `temp-ocr-${uuidv4()}-${file.originalname}`);
    fs.writeFileSync(tempFilePath, file.buffer);

    let extractedText = "";
    try {
      let pythonCmd = process.platform === 'win32' 
        ? path.join(__dirname, '../venv/Scripts/python.exe')
        : path.join(__dirname, '../venv/bin/python');
      
      if (!fs.existsSync(pythonCmd)) {
        pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
      }
      const cliPath = path.join(__dirname, '../ocr_cli.py');
      const { stdout, stderr } = await execPromise(`"${pythonCmd}" "${cliPath}" "${tempFilePath}"`);
      if (stderr && !stdout) {
        throw new Error(stderr);
      }
      extractedText = stdout.trim();
    } catch (ocrError) {
      console.error("PaddleOCR CLI failed:", ocrError.message);
      extractedText = "OCR extraction failed.";
    } finally {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    }

    // 3. AI Explanation using Ollama locally (placeholder if Ollama is not actually running, but implementing call)
    let aiSummary = "AI summary not available. Please ensure Ollama is running.";
    try {
      const response = await fetch(`${process.env.OLLAMA_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: process.env.OLLAMA_MODEL || 'gemma3:4b',
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
