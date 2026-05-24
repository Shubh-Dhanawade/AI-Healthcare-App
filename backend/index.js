require('dotenv').config();
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');

const authRoutes = require('./routes/authRoutes');
const prescriptionRoutes = require('./routes/prescriptionRoutes');
const insuranceRoutes = require('./routes/insuranceRoutes');
const chatbotRoutes = require('./routes/chatbotRoutes');

const app = express();

app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

app.use('/api/auth', authRoutes);
app.use('/api/prescriptions', prescriptionRoutes);
app.use('/api/insurance', insuranceRoutes);
app.use('/api/chat', chatbotRoutes);

app.get('/', (req, res) => {
  res.send('AI Healthcare App Backend Running');
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
