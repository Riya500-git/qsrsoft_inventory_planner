import React, { useState } from 'react';
import {
  Box,
  Text,
  Input,
  Button,
  ChakraProvider,
  Flex,
  Heading,
  Table,
  Tr,
  Td,
  Tbody,
  Thead,
  Th
} from '@chakra-ui/react';
import Typewriter from 'typewriter-effect';
import DatePicker from 'react-date-picker';
import 'react-date-picker/dist/DatePicker.css';
import 'react-calendar/dist/Calendar.css';

function MainPage() {
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [date, setDate] = useState(new Date());
  const [output, setOutput] = useState([]);
  const [dataExists, setDataExists] = useState(false);

  const handleRecommend = async () => {
    if (!latitude || !longitude || !date) {
      setOutput("Invalid inputs");
      return;
    }
    
    fetch(`http://127.0.0.1:5000/inventory?lat=${latitude}&lon=${longitude}&date=${date.toISOString().split('T')[0]}`, {mode:'cors'})
      .then((response) => response.json())
      .then((json) => {
        setOutput(json);
        setDataExists(true);
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
          {/* <Heading lineHeight="tall" color="yellow.500">
            <Highlight
              query="inventory"
              styles={{ px: '2', py: '1', rounded: 'full', bg: 'red.100' }}
            >
              Let's explore your inventory needs
            </Highlight> */}
            <Heading>
          <Text mt={6} fontSize="xl" color="green.800" fontWeight="bold">
            <Typewriter
              options={{
                strings: ["Let's explore your inventory needs"],
                autoStart: true,
                loop: true,
                delay: 50,
              }}
            />
          </Text>
          </Heading>

          <Flex mt={6} align="baseline">
            <Text fontSize="lg" textAlign="left" mr={2} color="green.800">
              Latitude:
            </Text>
            <Input
              type="text"
              placeholder="Enter latitude"
              value={latitude}
              onChange={(e) => setLatitude(e.target.value)}
              width="200px"
              style={{background:"white"}}
            />
            <Text fontSize="lg" textAlign="left" ml={4} mr={2} color="green.800">
              Longitude:
            </Text>
            <Input
              type="text"
              placeholder="Enter longitude"
              value={longitude}
              onChange={(e) => setLongitude(e.target.value)}
              width="200px"
              style={{background:"white"}}
            />
            <div style={{ padding: "20px"}}>
              <DatePicker onChange={setDate} value={date} />
            </div>
          </Flex>

          <Button mt={6} colorScheme="green" onClick={handleRecommend} width="300px">
            Recommend
          </Button>
          {dataExists ?
          <>
          <Table style={{borderWidth: "1px", borderColor: "black", tableLayout: "fixed", width: "300px", marginTop: "20px"}}>
            <Thead style={{borderColor: "black"}}>
              <Tr>
                <Th>Item</Th>
                <Th>Quantity</Th>
              </Tr>
            </Thead>
            <Tbody >
              {
                output.map((a) => {return (
                  <Tr>
                    <Td style={{borderColor: "black"}}>
                      {a[0]}
                    </Td>
                    <Td style={{borderColor: "black"}}>
                      {Math.abs(a[1])}
                    </Td>
                  </Tr>
                )})
              }
            </Tbody>
          </Table>
          <Text paddingTop="12" fontSize="sm" textAlign="left" mr={2} color="green.800">
            All data is fictional for the purpose of this project.
          </Text> </> : <></>}
          <Box bg="green.400" width="100%" height="30px" mt={6}></Box>

          <Text mt={6} fontSize="xl" fontWeight="bold" color="green.800">
            What is our goal?
            </Text>
          <Text mt={6} color="green.800">
            This tool aspires to help identify the number of ingredients needed for your franchise, based on location and date.
            <br/>
            <br/>
            We envision this tool can help you save resources and optimize the demand needed and reduce unnecessary consumption.
          </Text>
        </Box>
      </Box>
    </ChakraProvider>
  );
}

export default MainPage;
