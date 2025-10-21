import React from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import '../styles/Feedback.css'

export default function Feedback() {
  const location = useLocation();
  const navigate = useNavigate();

  // const { image, result } = location.state || {}; 
  const { image, result } = location.state || {
    result: {
      label: 'correct',
      exercise: 'Squat'
    },
    image: new File([], 'src\\pages\\63429_3.mp4', { type: 'video/mp4' }) // simulare video
  };
  const isCorrect = result?.label === 'correct';

  const improvementTips = {
    'Barbell row': ' eroare pentru barbell row',
    'Squat': ' eroare pentru squat',
    'Overhead press': ' Try to keep your knees locked and elbows fully extended at the top of the movement.'
  };


  return (
    <div className="home-container">
      <div className="logo-container"></div>

      <div className="content">
      <h1 className="title-feedback">Evaluation Result</h1>
        <div className='upload-box'>
        
        <div className={`result-box ${isCorrect ? 'correct' : 'incorrect'}`}>
          {isCorrect ? 'Correct execution' : (<>Incorrect execution <br /> {improvementTips[result.exercise]}</>)}
          
        </div>
        
        {image && (
          // <img
          //   src={URL.createObjectURL(image)}
          //   alt="Uploaded"
          //   className={`result-image ${isCorrect ? 'border-green' : 'border-red'}`}
          // />
          <video controls width="100%" className={`result-image ${isCorrect ? 'border-green' : 'border-red'}`}>
  <source src={URL.createObjectURL(image)} type="video/mp4" />
  Your browser does not support the video tag.
</video>
        )}

        <button className="home-button" onClick={() => navigate('/')}>
          Home
        </button>
        <ul className="benefits">
                <li>AI-based evaluation with instant feedback</li>
        </ul>
        <div className="footer">
                Powered by AI · FitPose 2025
        </div>
      </div>
      </div>
    </div>
  );
}
