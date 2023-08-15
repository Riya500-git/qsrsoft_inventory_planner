// import './App.css';
// import MainPage from './MainPage';
// import SecondPage from './SecondPage';

// function App() {
//   return (
//     <MainPage/>
//     // <SecondPage/>
//   );
// }

// export default App;


import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import MainPage from './MainPage';
import LoginPage from './LandingPage';
import SecondPage from './SecondPage';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/mainpage" element={<MainPage />} />
      </Routes>
    </Router>
  );
};

export default App;
