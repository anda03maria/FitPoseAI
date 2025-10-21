import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/HomePage.css';



export default function HomePage() {
    
   const [selectedExercise, setSelectedExercise] = useState('Squat');
   const navigate = useNavigate();

   const handleAddFile = () => {
    if (selectedExercise === 'Barbell row') {
        navigate('/upload-image', { state: { exercise: selectedExercise } });
    } else if (
         selectedExercise === 'Squat' || 
         selectedExercise === 'Overhead press' 
    ) {
      navigate('/upload-video', { state: { exercise: selectedExercise } });
      }
    };

    return (
        <div className="home-container">
        <div className="logo-container"></div>
        <div className="content">
            <h1 className="title-app"> FitPose <span className="title-ai">AI</span></h1>
            <div className="exercise-box">
            <p className="text-label">Choose one exercise:</p>
            <select
              value={selectedExercise}
              onChange={(e) => setSelectedExercise(e.target.value)}
              className="exercise-select"
            >
              <option>Squat</option>
              <option>Overhead press</option>
              <option>Barbell row</option>
            </select>
            <button
              onClick={handleAddFile} className="submit-btn">
              Load file
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
