import express from "express";
import { connectDB } from "./data/Connect.js";
import userRouter from "./routes/userroutes.js";
import cors from 'cors';

const app = express();
app.use(express.json());

const corsOptions = {
    origin: 'http://localhost:3000',  
    methods: ['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE'], 
    credentials: true
};
app.use(cors(corsOptions));

connectDB();

const PORT = 5001;
const server = app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

app.use('/api/auth', userRouter);

app.get('/', (req, res) => {
    res.send('hello');
});
