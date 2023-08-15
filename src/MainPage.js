import React, { useState, useEffect } from 'react';
import {
  Box,
  Text,
  Input,
  Button,
  ChakraProvider,
  Select,
  Flex,
  Heading,
  Highlight,
} from '@chakra-ui/react';
import Typewriter from 'typewriter-effect';

function MainPage() {
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [expectedDate, setExpectedDate] = useState('');
  const [output, setOutput] = useState('');

  const handleRecommend = async () => {
    if (!latitude || !longitude) {
      setOutput("Invalid inputs");
      return;
    }
    fetch("http://127.0.0.1:5000/inventory?location_id=1&item_id=1&week=1&month=1", {mode:'cors'})
      .then((response) => response.json())
      .then((json) => {
        setOutput(json);
      });
  };

  return (
    <ChakraProvider>
      <Box bg="green.100" minHeight="100vh" p={6}>
        <Box p={4} borderRadius="lg" boxShadow="md" bg="green.700">
          <Flex justify="space-between" align="center" mb={4}>
            <Text fontSize="xl" fontWeight="bold" color="white">
              QSRSoft Inventory Planner
            </Text>
            <Flex>
              <Button colorScheme="white" mr={2}>
                Home
              </Button>
              <Button colorScheme="white">About Us</Button>
            </Flex>
          </Flex>
        </Box>

        <Box ml={5} mr={5} mt={10}>
          <Heading lineHeight="tall" color="yellow.500">
            <Highlight
              query="eco-friendly"
              styles={{ px: '2', py: '1', rounded: 'full', bg: 'red.100' }}
            >
              Find an eco-friendly restaurant here today...
            </Highlight>
          </Heading>

          <Text fontSize="lg" textAlign="left" mt={6} color="green.800">
            Franchise
          </Text>
          <Select mt={2} placeholder="Select an option" width="300px">
            <option>Option 1</option>
            <option>Option 2</option>
            <option>Option 3</option>
          </Select>

          <Flex mt={6} align="baseline">
            <Text fontSize="lg" textAlign="left" mr={2} color="green.800">
              Latitude:
            </Text>
            <Input
              type="text"
              placeholder="Enter latitude"
              value={latitude}
              onChange={(e) => setLatitude(e.target.value)}
              width="300px"
            />
            <Text fontSize="lg" textAlign="left" ml={4} mr={2} color="green.800">
              Longitude:
            </Text>
            <Input
              type="text"
              placeholder="Enter longitude"
              value={longitude}
              onChange={(e) => setLongitude(e.target.value)}
              width="300px"
            />
          </Flex>

          <Button mt={6} colorScheme="green" onClick={handleRecommend} width="300px">
            Recommend
          </Button>

          <Text fontSize="lg" textAlign="left" mr={2} color="green.800">
            {output}
          </Text>

          <Text mt={6} fontSize="xl" color="green.800" fontWeight="bold">
            <Typewriter
              options={{
                strings: ['Find an eco-friendly restaurant here today...'],
                autoStart: true,
                loop: true,
              }}
            />
          </Text>

          <Box bg="green.400" width="100%" height="30px" mt={6}></Box>

          <Text mt={6} fontSize="xl" fontWeight="bold" color="green.800">
            What is Our goal?          
            </Text>
          <Text mt={6} color="green.800">
          Lorem ipsum is typically a corrupted version of De finibus bonorum et malorum, a 1st-century BC text by the Roman statesman and philosopher Cicero, with words altered, added, and removed to make it nonsensical and improper Latin.
          Lorem ipsum is typically a corrupted version of De finibus bonorum et malorum, a 1st-century BC text by the Roman statesman and philosopher Cicero, with words altered, added, and removed to make it nonsensical and improper Latin.
          </Text>
        </Box>
      </Box>
    </ChakraProvider>
  );
}

export default MainPage;
