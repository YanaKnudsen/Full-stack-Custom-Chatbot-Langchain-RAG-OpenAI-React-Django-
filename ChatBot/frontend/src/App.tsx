
import "./App.css";
import {Routes,Route} from "react-router-dom";
import HomePage from "./HomePage.tsx";



function App() {

  return (
    <>
        <div>
            <Routes>
                <Route path="" element={<HomePage/>}/>
                {/*<Route path="/about" element={<Services/>}/>*/}
            </Routes>

        </div>
    </>
  )
}

export default App
