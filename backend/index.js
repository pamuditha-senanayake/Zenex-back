import express from "express";
import bodyParser from "body-parser";
import cors from 'cors';
import dotenv from "dotenv";

import db from './db.js';
import userManagementController from "./controllers/userManagement.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3003;

const allowedOrigins = [process.env.FRONTEND_URL || 'http://localhost:3000'];

app.use(cors({
    origin: function(origin, callback) {
        if (!origin) return callback(null, true); // allow requests like Postman or server-to-server
        if (allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            callback(new Error('CORS policy: Not allowed origin'));
        }
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    credentials: true
}));

// Explicitly handle OPTIONS preflight requests
app.options('*', cors());

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
