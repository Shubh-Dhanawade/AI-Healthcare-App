const express = require('express');
const { uploadInsurance, getInsuranceDocs } = require('../controllers/insuranceController');
const { protect } = require('../middlewares/authMiddleware');
const upload = require('../middlewares/uploadMiddleware');

const router = express.Router();

router.post('/upload', protect, upload.single('file'), uploadInsurance);
router.get('/', protect, getInsuranceDocs);

module.exports = router;
