import mongoose from "mongoose";

const assetschema = new mongoose.Schema({
    symbol:{
        type : String,
        required : true
    },
    name : {
        type : String,
        required : true
    },
    purchasedPrice : {
        type : Number,
        required : true
    },
    currentPrice : {
        type : Number,
        required : true
    },
    quantity : {
        type : Number,
        required : true
    }
},{_id : false});

const userschema = new mongoose.Schema({
    username : {
        type : String,
        required : true
    },
    email : {
        type : String,
        required : true
    },
    password : {
        type : String,
        required : true
    },
    aadhar :{
        type : String,
        required : true
    },
    pan :{
        type : String,
        required : true
    },
    stocks : {
        type : [assetschema],
        required : true
    },
    cryptos : {
        type : [assetschema],
        required : true
    }
});

export const user = mongoose.model('user',userschema);