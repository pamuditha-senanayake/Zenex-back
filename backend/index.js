import express from "express";
import bodyParser from "body-parser";
import cors from 'cors';
import dotenv from "dotenv";

import db from './db.js';
import userManagementController from "./controllers/userManagement.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3003;

// ✅ CORS middleware — allow frontend URL from env or localhost
app.use(cors({
    origin: [process.env.FRONTEND_URL || 'http://localhost:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    credentials: true
}));

// ✅ Body parsers
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// ✅ EJS setup
app.set("view engine", "ejs");

// ✅ Make env URLs available to EJS templates (optional)
app.use((req, res, next) => {
    res.locals.apiBaseUrl = process.env.BACKEND_URL || `http://localhost:${port}`;
    res.locals.frontendUrl = process.env.FRONTEND_URL || 'http://localhost:3000';
    next();
});

// ✅ Routes
app.use("/", userManagementController);

// ✅ DB connection check
db.query("SELECT 1")
    .then(() => console.log('DB connection succeeded.'))
    .catch(err => console.log('DB connection failed.\n' + err));

// ✅ Start server
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
