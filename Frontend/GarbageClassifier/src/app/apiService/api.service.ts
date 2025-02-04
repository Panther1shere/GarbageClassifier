import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = "http://localhost:8000/predict/";
  constructor() { }



  async identifyWaste(file: File): Promise<string> {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(this.apiUrl, {
        method: "POST",
        body: formData,
        mode: "cors",
        headers: {

        }
      } as RequestInit); // Explicitly casting to RequestInit

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.predicted_class || "Unknown";

    } catch (error) {
      console.error("Error identifying object:", error);

      // Handle specific errors
      if (error instanceof TypeError) {
        return "Network error or CORS issue";
      } else if (error instanceof SyntaxError) {
        return "Invalid response from server";
      } else {
        return "Error identifying waste";
      }
    }
  }

}
