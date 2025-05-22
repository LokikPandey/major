import mongoose from "mongoose";


const PATH = 'mongodb://localhost:27017/finance';

export const connectDB = async () => {
    mongoose.connect(PATH,{
    })
    .then(() => console.log("Connected to MongoDB"))
    .catch((e)=>console.log(e.message))
}