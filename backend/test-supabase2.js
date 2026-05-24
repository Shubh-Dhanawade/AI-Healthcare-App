require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');

async function testConnection() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_KEY;

  let result = {};
  
  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    const { data, error } = await supabase.from('non_existent_table').select('*').limit(1);
    result = { data, error };
  } catch (err) {
    result = { exception: err.message };
  }
  
  fs.writeFileSync('output.json', JSON.stringify(result, null, 2));
}

testConnection();
