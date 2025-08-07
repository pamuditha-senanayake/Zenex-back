import express from "express";
import bodyParser from "body-parser";
import cors from 'cors';
import dotenv from "dotenv";

import db from './db.js';
import userManagementController from "./controllers/userManagement.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3003;

// Define allowed origins, from env or fallback
const allowedOrigins = [
    process.env.FRONTEND_URL || 'http://localhost:3000',
    'http://localhost:3003' // you can add others here if needed
];

app.use(cors({
    origin: allowedOrigins,
    credentials: true
}));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.set("view engine", "ejs");

app.use((req, res, next) => {
    res.locals.apiBaseUrl = process.env.BACKEND_URL || `http://localhost:${port}`;
    res.locals.frontendUrl = process.env.FRONTEND_URL || 'http://localhost:3000';
    next();
});

app.use("/", userManagementController);

db.query("SELECT 1")
    .then(() => console.log('DB connection succeeded.'))
    .catch(err => console.log('DB connection failed.\n' + err));

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
