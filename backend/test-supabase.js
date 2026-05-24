require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

async function testConnection() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_KEY;

  console.log('Testing connection to URL:', supabaseUrl);
  
  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    // Simple query to check connection. Using auth health or a dummy table
    const { data, error } = await supabase.from('non_existent_table').select('*').limit(1);
    
    // We expect an error about table not existing, but NOT a network/URL error if connected
    if (error && error.code === 'PGRST116') { // PGRST116 is usually table not found, which means connection is SUCCESSFUL
       console.log('Connection successful, but table not found (as expected).');
    } else if (error && error.message && error.message.includes('fetch')) {
       console.log('Connection Failed (Network Error):', error.message);
    } else {
       console.log('Connection response:', { data, error });
    }
  } catch (err) {
    console.error('Connection Failed (Exception):', err.message);
  }
}

testConnection();
