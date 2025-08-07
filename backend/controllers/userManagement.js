import express from 'express';
import bcrypt from 'bcrypt';
import pool from '../db.js';

const router = express.Router();

// Register route
router.post('/register', async (req, res) => {
    const { username, password } = req.body;
    try {
        const userCheck = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        if (userCheck.rows.length > 0) {
            return res.status(400).json({ error: 'Username already exists' });
        }
        const hashedPassword = await bcrypt.hash(password, 10);
        await pool.query('INSERT INTO users (username, password) VALUES ($1, $2)', [username, hashedPassword]);
        res.json({ message: 'User registered successfully' });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Login route
router.post('/login', async (req, res) => {
    const { username, password } = req.body;
    console.log(`[Login] Received login attempt for username: ${username}`);

    try {
        const result = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        console.log(`[Login] User lookup result rows count: ${result.rows.length}`);

        if (result.rows.length === 0) {
            console.log('[Login] No user found with that username');
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const user = result.rows[0];
        const valid = await bcrypt.compare(password, user.password);
        console.log(`[Login] Password valid: ${valid}`);

        if (!valid) {
            console.log('[Login] Password mismatch');
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        console.log('[Login] Login successful');
        res.json({ message: 'Login successful' }); // must be here for success
    } catch (err) {
        console.error('[Login] Internal server error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});


export default router;
