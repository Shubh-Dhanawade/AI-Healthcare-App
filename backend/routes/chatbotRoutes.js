const express = require('express');
const { chatWithAI, getChatHistory } = require('../controllers/chatbotController');
const { protect } = require('../middlewares/authMiddleware');

const router = express.Router();

router.post('/', protect, chatWithAI);
router.get('/', protect, getChatHistory);

module.exports = router;
