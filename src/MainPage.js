import React, {useState} from 'react';
import { Box, Center, Text, Input, Button, ChakraProvider, Select } from '@chakra-ui/react';

function MainPage() {
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');

  const handleRecommend = async () => {
    try {
    } catch (error) {
      console.error('Error', error);
    }
  };

  return (
    <ChakraProvider>
      <Center h="100vh">
        <Box bg="rgba(255, 255, 255, 0.7)" p={8} borderRadius="lg" boxShadow="xl">
          <Center>
            <img src="logo.png" alt="CloudX Hackathon" />
          </Center>
          <Text mt={4} textAlign="left">Franchise</Text>
          <Select mt={2} placeholder="Select an option">
            <option>Option 1</option>
            <option>Option 2</option>
            <option>Option 3</option>
          </Select>
          {/* <Text mt={4} textAlign="left">Country</Text> */}
          {/* <Input type="text" mt={2} placeholder="Search and select a country" /> */}
          {/* <Select mt={2} placeholder="Select a country">
            {countries.map((country, index) => (
              <option key={index}>{country}</option>
            ))}
          </Select> */}
          <Text mt={4} textAlign="left">Latitude</Text>
          <Input
            type="text"
            mt={2}
            placeholder="Enter latitude"
            value={latitude}
            onChange={(e) => setLatitude(e.target.value)}
          />
          <Text mt={2} textAlign="left">Longitude</Text>
          <Input
            type="text"
            mt={2}
            placeholder="Enter longitude"
            value={longitude}
            onChange={(e) => setLongitude(e.target.value)}
          />
          <Button mt={4} colorScheme="blue" onClick={handleRecommend}>Recommend</Button>
        </Box>
      </Center>
    </ChakraProvider>
  );
}

export default MainPage;
