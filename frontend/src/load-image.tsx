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
  MenuItem,
  Card,
} from "@mui/material";
import axios from "axios";
import models, { Model, parseBreed } from "./models";

const LoadImage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<string[] | null>(null);
  const [model, setModel] = useState<Model | null>(null);

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
    setResult(null);
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/recognize-dog-breed/" + model?.value,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("File uploaded successfully:", response.data);
      setResult(response.data);
      console.log(response.data);
      setLoading(false);
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
        py={2}
        mt={3}
        width={"100%"}
      >
        <>
          <Card
            sx={{
              width: "100%",
              padding: 2,
              display: "flex",
              flexDirection: "column",
              rowGap: 2,
              mx: "auto",
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
              {model && (
                <Typography>
                  Dog breeds that are recognized by the model:{" "}
                  {model.breeds.map((breed) => parseBreed(breed)).join(", ")}
                </Typography>
              )}
              <TextField
                select
                label="Model"
                value={model?.value || ""}
                onChange={(event) => {
                  const selectedModel = models.find(
                    (model) => model.value === event.target.value
                  );
                  setModel(selectedModel || null);
                }}
              >
                {models.map((model) => (
                  <MenuItem key={model.value} value={model.value}>
                    {model.label}
                  </MenuItem>
                ))}
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
        {loading && (
          <>
            <CircularProgress sx={{ marginTop: 5, marginBottom: 2 }} />{" "}
            <Typography variant="body1">Getting result</Typography>
          </>
        )}
      </Box>
      <Box display={"flex"}>
        {preview && !loading && (
          <Box
            mt={2}
            display={"flex"}
            flexDirection={"column"}
            alignItems={"center"}
            width={"100%"}
          >
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
                    <TableCell align="right">Probability</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.map((item) => (
                    <TableRow key={item[0]}>
                      <TableCell>{parseBreed(item[0])}</TableCell>
                      <TableCell align="right">
                        {parseFloat(item[1]).toPrecision(2)}
                      </TableCell>
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
