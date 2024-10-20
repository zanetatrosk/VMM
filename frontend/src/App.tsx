import Navbar from './navbar';
import LoadImage from './load-image';
import { Container, createTheme, CssBaseline, ThemeProvider } from '@mui/material';

const theme = createTheme({
  breakpoints: {
    values: {
      mobile: 0,
      tablet: 640,
      laptop: 940,
      desktop: 1200,
    },
  },
});

declare module '@mui/material/styles' {
  interface BreakpointOverrides {
    xs: false; // removes the `xs` breakpoint
    sm: false;
    md: false;
    lg: false;
    xl: false;
    mobile: true; // adds the `mobile` breakpoint
    tablet: true;
    laptop: true;
    desktop: true;
  }
}


function App() {
  return (
    <>
    <ThemeProvider theme={theme}>
    <CssBaseline />
      <Navbar/>
      <Container maxWidth={"laptop"}>
      <LoadImage/>
      </Container>
      </ThemeProvider>
    </>
  );
}

export default App;
