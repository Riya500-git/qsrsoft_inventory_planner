// import React from 'react';
// import { AuthProvider } from '@descope/react-sdk';
// import { useHistory } from 'react-router-dom';
// import MainPage from './MainPage';
// import {Descope, SignUpOrInFlow } from '@descope/react-sdk';

// const LoginPage = () => {
//   const history = useHistory();

//   const handleSuccess = (e) => {
//     console.log(e.detail.user.name)
// 	console.log(e.detail.user.email)
//     history.push('/mainpage'); 
//   };

//   const handleError = (error) => {
//     console.log('Error!', error);
//   };

//   return (
//     <AuthProvider projectId="P2U1pXwJAJkC8C4AEcVqMyyYUCtP">
//       <Descope
//         flowId="sign-up-or-in"
//         theme="light"
//         onSuccess={handleSuccess}
//         onError={handleError}
//       />
//     </AuthProvider>
//   );
// };

// export default LoginPage;

// Use the code below to add the session validation to your server application.
// More info: https://docs.descope.com/build/guides/session/#Go
// authorized, userToken, err :=
// 	descopeClient.Auth.ValidateSessionWithToken(sessionToken)


import React from 'react';
import { AuthProvider } from '@descope/react-sdk';
import { useNavigate } from 'react-router-dom'; 
import { Descope } from '@descope/react-sdk';
import { ChakraProvider, Box, Center } from '@chakra-ui/react';

const LoginPage = () => {
  const navigate = useNavigate(); 

  const handleSuccess = (e) => {
    console.log(e.detail.user.name);
    console.log(e.detail.user.email);
    navigate('/mainpage'); 
  };

  const handleError = (error) => {
    console.log('Error!', error);
  };

  return (
    <ChakraProvider>
    <Center mt={40}>
    <Box className="info" maxWidth="500px">
    <AuthProvider projectId="P2U1pXwJAJkC8C4AEcVqMyyYUCtP">
      <Descope
        flowId="sign-up-or-in"
        theme="light"
        onSuccess={handleSuccess}
        onError={handleError}
      />
    </AuthProvider>
    </Box>
    </Center>
    </ChakraProvider>
  );
};

export default LoginPage;
