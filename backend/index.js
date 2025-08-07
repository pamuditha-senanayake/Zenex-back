import express from "express";
import bodyParser from "body-parser";
import cors from 'cors';
import dotenv from "dotenv";

import db from './db.js';
import userManagementController from "./controllers/userManagement.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3003;

app.use(cors({
    origin: ['http://localhost:3000'],
    credentials: true
}));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use("/", userManagementController);

db.query("SELECT 1")
    .then(() => console.log('DB connection succeeded.'))
    .catch(err => console.log('DB connection failed.\n' + err));

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
