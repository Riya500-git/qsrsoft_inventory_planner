import React from 'react';
import { Box, Center, Text, Table, Thead, Tbody, Tr, Th, Td, ChakraProvider } from '@chakra-ui/react';

function SecondPage() {
  return (
    <ChakraProvider>
      <Center h="100vh">
        <Box display="flex">
          {/* <img src="logo.png" alt="Logo" width="100px" height="100px" /> */}
          <Box bg="rgba(255, 255, 255, 0.7)" p={4} borderRadius="md" boxShadow="md">
            <Text fontSize="xl" fontWeight="bold">Your Recommendations</Text>
            <Table mt={4} variant="simple">
              <Thead>
                <Tr>
                  <Th>Item</Th>
                  <Th>Quantity</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td></Td>
                  <Td></Td>
                </Tr>
                <Tr>
                  <Td></Td>
                  <Td></Td>
                </Tr>
                <Tr>
                  <Td></Td>
                  <Td></Td>
                </Tr>
              </Tbody>
            </Table>
          </Box>
        </Box>
      </Center>
    </ChakraProvider>
  );
}

export default SecondPage;
