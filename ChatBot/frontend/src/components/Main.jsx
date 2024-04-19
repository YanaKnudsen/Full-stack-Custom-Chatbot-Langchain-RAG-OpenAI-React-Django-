import {motion} from "framer-motion";
import  './main.scss'
import backgroundVideo from "../../../qnudsenapp/src/assets/homePageVideo.mp4"
import Navbar from "./navbar/Navbar.jsx";
import MainPageText from "./MainPageText.jsx";
import {theme,desriptiontheme} from "../themes/theme.tsx"
import {createTheme, responsiveFontSizes, ThemeProvider, Typography} from "@mui/material";

const Main=()=>{
return(
    <div className="mainPage">
   <video src={backgroundVideo} autoPlay loop muted/>
        <div className="nestedContent">
            {/*<Navbar/>*/}
            <MainPageText/>
            <div className="bottomBar">
<div className="textBottomBar">

                <ThemeProvider theme={desriptiontheme}>
                    <Typography variant="h5" gutterBottom>Learn more about our sercices</Typography>
                </ThemeProvider>
</div>
            </div>

        </div>
    </div>
)
}
export default Main