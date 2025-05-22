import { user } from "../models/usermodel.js";
import bcrypt from "bcryptjs";

export const register = async (req, res, next) => {
    try {
        const { username, email, password, aadhar, pan } = req.body;
        const usercheck = await user.findOne({ email });
        if (usercheck) {
            console.log("User already exists");
            return res.json({ message: "User already exists", status: false });
        }
        const newpass = await bcrypt.hash(password, 10);
        const usercreate = await user.create({ username, email, password: newpass, aadhar, pan });

        console.log("User created successfully");
        return res.json({ 
            message: "User created successfully",
            status: true ,
            user:{
            _id:usercreate._id,
            username: usercreate.username,
            email: usercreate.email,
            }
        });
    } catch (e) {
        console.log(e.message);
        next(e);
    }
};

export const login = async (req, res, next) => {
    try {
        const { email, password } = req.body;
        const usercheck = await user.findOne({ email });

        if (!usercheck) {
            console.log("User not found");
            return res.json({ message: "User not found", status: false });
        }

        const isMatch = await bcrypt.compare(password, usercheck.password);
        if (!isMatch) {
            console.log("Incorrect password");
            return res.json({ message: "Incorrect password", status: false });
        }

        console.log("Login successful");
        return res.json({
            message: "Login successful",
            status: true,
            user: {
                _id: usercheck._id,
                username: usercheck.username,
                email: usercheck.email
            }
        });

    } catch (e) {
        console.log(e.message);
        next(e);
    }
};

export const getport = async (req, res, next) => {
    try{
        const User = await user.findById(req.params.id);
        if(!User){
            return res.status(404).json({message:"user not found"});
        }
        return res.json({message:"user found",
            stocks:User.stocks || [],
            cryptos:User.cryptos || []
        });
    }catch(e){
        console.log(e.message);
        return res.status(500).json({message:"Internal server error"});
    }
};

export const getuser = async(req,res,next)=>{
    try
    {
        const User  =  await user.findById(req.params.id);
        if(!User)
        {
            return res.status(404).json({message:"user not found"});
        }

        return res.json({message:"user found",
            username:User.username,
            email:User.email,
            aadhar:User.aadhar,
            pan:User.pan,
            number:User.number
        });
    }catch(e){
        console.log(e.message);
        return res.status(500).json({message:"Internal server error"});
    }
    
}

export const addstock = async (req, res, next) => {
    const { stocks } = req.body; // expects an array of stocks to be added
    try {
        const userData = await user.findById(req.params.id);
        if (!userData) {
            return res.status(404).json({ message: "User not found" });
        }

        // If stocks is an array, push all items one by one
        if (Array.isArray(stocks)) {
            userData.stocks.push(...stocks);
        } else {
            // if it's a single object, push it directly
            userData.stocks.push(stocks);
        }

        await userData.save();
        return res.json({ message: "Stocks added successfully" });
    } catch (e) {
        console.log(e.message);
        return res.status(500).json({ message: "Internal server error" });
    }
};

export const addcrypto = async (req, res, next) => {
    const { cryptos } = req.body;

    try {
        const User = await user.findById(req.params.id);
        if (!User) {
            console.log("User not found");
            return res.status(404).json({ message: "User not found" });
        }

        if (Array.isArray(cryptos)) {
            User.cryptos.push(...cryptos);
        } else {
            User.cryptos.push(cryptos);
        }
        await User.save();

        return res.json({ message: "Crypto added successfully" });
    } catch (e) {
        console.log(e.message);
        return res.status(500).json({ message: "Internal server error" });
    }
};

export const deletestock = async (req, res, next) => {
  try {
    const User = await user.findById(req.params.id);
    const { symbol, type } = req.body;  // type is expected: 'stock' or 'crypto'

    if (!User) {
      return res.status(404).json({ message: "User not found" });
    }

    if (!type || (type !== "stock" && type !== "crypto")) {
      return res.status(400).json({ message: "Invalid asset type" });
    }

    // Select the correct array to delete from
    const assetArray = type === "stock" ? User.stocks : User.cryptos;

    // Find and remove asset with the given symbol
    const index = assetArray.findIndex(asset => asset.symbol === symbol);

    if (index === -1) {
      return res.status(404).json({ message: `${type} not found` });
    }

    assetArray.splice(index, 1);
    await User.save();

    return res.json({ message: `${type} deleted successfully` });

  } catch (e) {
    console.log(e.message);
    return res.status(500).json({ message: "Internal server error" });
  }
};