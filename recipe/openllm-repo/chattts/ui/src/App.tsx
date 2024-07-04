import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Input, InputNumber, Button, message, Layout, Typography } from 'antd';
import 'antd/dist/reset.css';

const { Content } = Layout;
const { Title } = Typography;

const App: React.FC = () => {
  const [textInput, setTextInput] = useState<string>('Chat T T S 是一个开源的语音转文字技术，能够很好的还原说话人的语气和停顿。输入你的文字开始尝试吧。');
  const [numberInput, setNumberInput] = useState<number>(2);
  const [audioSrc, setAudioSrc] = useState<string | null>(null);

  useEffect(() => {
    // Clean up URL object when component unmounts or before creating a new one
    return () => {
      if (audioSrc) {
        URL.revokeObjectURL(audioSrc);
      }
    };
  }, [audioSrc]);

  const handleTextInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTextInput(event.target.value);
  };

  const handleNumberInputChange = (value: number | null) => {
    if (value !== null) {
      setNumberInput(value);
    }
  };

  const handleSubmit = async () => {
    if (audioSrc) {
      URL.revokeObjectURL(audioSrc); // Revoke the previous audio URL
      setAudioSrc(null); // Reset the audio source to trigger re-render
    }

    try {
      const response = await axios.post('/tts', {
        text: textInput,
        speaker: numberInput,
      }, {
        responseType: 'blob',
      });

      const audioUrl = URL.createObjectURL(new Blob([response.data], { type: 'audio/mp3' }));
      setAudioSrc(audioUrl);
      message.success('Audio fetched successfully!');
    } catch (error) {
      console.error('Error fetching audio:', error);
      message.error('Failed to fetch audio');
    }
  };

  return (
    <Layout style={{ minHeight: '100vh', padding: '50px' }}>
      <Content style={{ background: '#fff', padding: '24px', margin: 0, minHeight: 280 }}>
        <Title level={2}>ChatTTS</Title>
        <div style={{ marginBottom: '16px' }}>
          <label>
            Text Input:
            <Input.TextArea value={textInput} onChange={handleTextInputChange} style={{ width: '300px', marginLeft: '10px' }} />
          </label>
        </div>
        <div style={{ marginBottom: '16px' }}>
          <label>
            Number Input:
            <InputNumber value={numberInput} onChange={handleNumberInputChange} style={{ width: '300px', marginLeft: '10px' }} />
          </label>
        </div>
        <Button type="primary" onClick={handleSubmit}>Submit</Button>
        {audioSrc && (
          <div style={{ marginTop: '20px' }}>
            <audio controls style={{ width: '100%' }}>
              <source src={audioSrc} type="audio/mp3" />
              Your browser does not support the audio element.
            </audio>
            <Button type="link" href={audioSrc} download="audio.mp3">Download Audio</Button>
          </div>
        )}
      </Content>
    </Layout>
  );
};

export default App;
