import Navbar from './navbar';
import LoadImage from './load-image';
import { Container } from '@mui/material';

function App() {
  return (
    <>
      <Navbar/>
      <Container maxWidth={"md"}>
      <LoadImage/>
      </Container>
    </>
  );
}

export default App;
