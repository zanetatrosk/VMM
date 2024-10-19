import React, { useEffect, useState } from "react";
import {
  Button,
  TextField,
  Box,
  Typography,
  CircularProgress,
  Table,
  TableContainer,
  TableHead,
  TableCell,
  TableBody,
  TableRow,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
} from "@mui/material";
import axios from "axios";

const LoadImage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<string[] | null>(null);
  const [model, setModel] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setResult(null);
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  useEffect(() => {
    if (result && result.length > 0) {
      document.getElementById("myid")?.scrollIntoView({ behavior: "smooth" });
    }
  }, [result]);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/recognize-dog-breed",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("File uploaded successfully:", response.data);
      setResult(response.data.predictions);
      console.log(response.data);
      setLoading(false);
      document.getElementById("myid")?.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      console.error("Error uploading file:", error);
      setLoading(false);
    }
  };

  return (
    <>
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        p={2}
        mt={3}
      >
          <>
            <Card
              sx={{
                width: "100%",
                padding: 2,
                display: "flex",
                flexDirection: "column",
                rowGap: 2,
              }}
            >
              <Typography variant="h5">Upload Image</Typography>
              <Box
                display={"flex"}
                justifyContent={"space-between"}
                flexDirection={"column-reverse"}
                width={"100%"}
                rowGap={2}
              >
                <TextField
                  type="file"
                  onChange={handleFileChange}
                  inputProps={{ accept: "image/*" }}
                />
                <TextField
                  select
                  label="Model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <MenuItem value="mobilenet">MobileNet</MenuItem>
                  <MenuItem value="resnet">ResNet</MenuItem>
                </TextField>
              </Box>
              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                disabled={!selectedFile || !model || loading}
                sx={{ mt: 2 }}
              >
                Upload
              </Button>
            </Card>
          </>
        {loading && <CircularProgress sx={{ marginTop: 5 }}/>}
      </Box>
      <Box display={"flex"}>
        {preview && !loading && (
          <Box mt={2} display={"flex"} flexDirection={"column"} alignItems={"center"} width={"100%"}>
            <Typography variant="h5" gutterBottom>
              Preview
            </Typography>
            <img
              src={preview}
              alt="Preview"
              style={{ maxHeight: "40rem", maxWidth: "40rem" }}
            />
          </Box>
        )}
        {result && (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            p={2}
            pr={0}
            id="myid"
            width={"100%"}
          >
            <Typography variant="h5" gutterBottom>
              Result
            </Typography>
            <TableContainer component={Paper} sx={{ maxWidth: "50rem" }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Breed</TableCell>
                    <TableCell>Probability</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.map((item, index) => (
                    <TableRow>
                      <TableCell key={index}>{item.split(":")[0]}</TableCell>
                      <TableCell key={index}>{item.split(":")[1]}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Box>
    </>
  );
};

export default LoadImage;
