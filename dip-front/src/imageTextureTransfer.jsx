import React, { useState } from 'react';
import styled from 'styled-components';

const Container = styled.div`
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
    color: #333;
    padding: 20px;
    text-align: center;
`;

const Title = styled.h1`
    margin-bottom: 20px;
    color: #2c3e50;
`;

const Form = styled.form`
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    max-width: 400px;
    margin: 0 auto;
`;

const Label = styled.label`
    display: block;
    margin-bottom: 8px;
    font-weight: bold;
`;

const Input = styled.input`
    display: block;
    margin-bottom: 15px;
`;

const Button = styled.button`
    background-color: #2980b9;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    width: 100%;

    &:hover {
        background-color: #3498db;
    }
`;

const OutputImage = styled.img`
    display: block;
    margin: 20px auto;
    border: 2px solid #2980b9;
    border-radius: 5px;
    width: 300px; /* Set a fixed width */
    height: 300px; /* Set a fixed height */
    object-fit: cover; /* Ensures the image covers the area without stretching */
    max-width: 100%; /* Ensures it does not overflow the container */
    max-height: 100%; /* Ensures it does not overflow the container */
`;

const ImageTextureTransfer = () => {
    const [outputUrl, setOutputUrl] = useState('');
    const [inputUrl, setInputUrl] = useState('');
    const [targetUrl, setTargetUrl] = useState('');
    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        const inputImage = event.target.input_image.files[0];
        const targetImage = event.target.target_image.files[0];
    
        
        setInputUrl(URL.createObjectURL(inputImage));
        setTargetUrl(URL.createObjectURL(targetImage));
    
        formData.append('input_image', inputImage);
        formData.append('target_image', targetImage);
    
        try {
            const response = await fetch('http://127.0.0.1:5000/upload', {
                method: 'POST',
                body: formData
            });
    
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                setOutputUrl(url);
            } else {
                const errorData = await response.json();
                alert(`Error: ${errorData.error}`);
            }
        } catch (error) {
            alert('Network error occurred: ' + error.message);
        }
    };

   

    return (
        <Container>
            <Title>Texture Transfer Image Processing</Title>
            <Form onSubmit={handleSubmit}>
                <Label htmlFor="input-image">Input Image:</Label>
                <Input type="file" id="input-image" name="input_image" accept="image/*" required />
                <Label htmlFor="target-image">Target Image:</Label>
                <Input type="file" id="target-image" name="target_image" accept="image/*" required />
                <Button type="submit">Process Images</Button>
            </Form>
            <h2>Images:</h2>
            <div>
                <h3>Input Image:</h3>
                {inputUrl && <OutputImage src={inputUrl} alt="Input" />}
            </div>
            <div>
                <h3>Target Image:</h3>
                {targetUrl && <OutputImage src={targetUrl} alt="Target" />}
            </div>
            <div>
                <h3>Output Image:</h3>
                {outputUrl && <OutputImage src={outputUrl} alt="Output" />}
            </div>
        </Container>
    );
};

export default ImageTextureTransfer;
