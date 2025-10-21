import axios from 'axios';
import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import '../styles/UploadImage.css';



export default function UploadImage() {
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

  const handleSubmit = async (modelType) => {
    const formData = new FormData();
    formData.append("file", file);  
    formData.append("model_type", modelType);
    formData.append('exercise', selectedExercise);

    if (!file) return alert('Choose an image from files.');
    if (!selectedExercise) return alert('No exercise selected.');
    
    try {
        const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResult(response.data);
        console.log("RESPONSE DATA", response.data);
        navigate('/feedback', {
          state: {
            image: file,
            result: {
              ...response.data,
              exercise: selectedExercise
            }
          }
        });
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Failed to upload and evaluate the image.');
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
            accept="image/*"
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
              <img src={previewUrl} alt="Preview" className="preview-image" />
              <div className="remove-wrapper">
                <button onClick={handleRemoveFile} className="remove-btn">Remove file</button>
              </div>            
            </div>
          )}

          <div className="model-button-group">
            <button className="model-button" onClick={() => handleSubmit("image")}>
              Evaluate with CNN model (.h5)
            </button>
            <button className="model-button" onClick={() => handleSubmit("landmark")}>
              Evaluate with XGBoost model (.joblib)
            </button>
          </div>

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
