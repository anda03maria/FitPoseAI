import axios from 'axios';
import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import '../styles/UploadVideo.css';


export default function UploadVideo() {
  const location = useLocation();
  const selectedExercise = location.state?.exercise;
  const [file, setFile] = useState(null);
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected)); 
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
  };
  
  const handleSubmit = async () => {
    if (!file) return alert('Select a video file.');
    if (!selectedExercise) return alert('No exercise selected.');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('exercise', selectedExercise);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
      navigate('/feedback', {
        state: {
          video: file,
          result: {
            ...response.data,
            exercise: selectedExercise
          }
        }
      });
    } catch (error) {
      console.error('Upload failed', error);
      alert('Failed to upload and evaluate the video.');
    }

  };

  return (
    <div className="home-container">
    <div className="logo-container"></div>

    <div className="content">
      <h1 className="title-exercise">{selectedExercise}</h1>
      <div className="upload-box">
      <p className="text-label">Upload a file:</p>
        {!file && (
          <>
            <input
              type="file"
              id="upload-file"
              accept="video/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            <label htmlFor="upload-file" className="custom-upload-button">
                Choose file
            </label>
          </>
        )}

          {previewUrl && (
            <div className="preview-section">
              <video src={previewUrl} alt="Preview" controls className="preview-video" />
              <div className="remove-wrapper">
                <button onClick={handleRemoveFile} className="remove-btn">Remove file</button>
              </div>            
            </div>
          )}
      
      <button
        onClick={handleSubmit} className="upload-button">
        Start evaluation
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
