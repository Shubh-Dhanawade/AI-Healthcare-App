const express = require('express');
const { uploadPrescription, getPrescriptions } = require('../controllers/prescriptionController');
const { protect } = require('../middlewares/authMiddleware');
const upload = require('../middlewares/uploadMiddleware');

const router = express.Router();

router.post('/upload', protect, upload.single('file'), uploadPrescription);
router.get('/', protect, getPrescriptions);

module.exports = router;
