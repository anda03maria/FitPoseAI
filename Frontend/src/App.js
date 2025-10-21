import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import  HomePage from './pages/HomePage';
import UploadImage from './pages/UploadImage';
import UploadVideo from './pages/UploadVideo';
import Feedback from './pages/Feedback';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload-image" element={<UploadImage />} />
        <Route path="/upload-video" element={<UploadVideo />} />
        <Route path="/feedback" element={<Feedback />} />
      </Routes>
    </Router>
  );
}

export default App;
