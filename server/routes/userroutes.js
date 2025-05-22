import { register, login ,getport,addstock,addcrypto,getuser,deletestock} from "../controllers/usercontrollers.js";
import express from "express";
const router = express.Router()

router.post('/register',register);
router.post('/login',login);
router.get('/getuser/:id',getuser);
router.get('/getport/:id',getport);
router.post('/addstock/:id',addstock);
router.post('/addcrypto/:id',addcrypto);
router.delete('/deletestock/:id',deletestock);
export default router;