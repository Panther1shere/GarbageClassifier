import { Component } from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {HttpClient, HttpClientModule} from '@angular/common/http';
import {MatButton} from '@angular/material/button';
import {NgForOf, NgIf, NgStyle} from '@angular/common';
import {MatProgressSpinner} from '@angular/material/progress-spinner';
import {MatIcon} from '@angular/material/icon';
import {ApiService} from '../apiService/api.service';


/**
 * Author: Kumawat Mohit, Fulara Utkarsh
 * Date: 27-02-2025
 * Purpose: This is the Typescript file for the home component. Consists of all the methods for the home component.
 * All the logic and request to backend is sent from here
 */


@Component({
  selector: 'app-home',
  imports: [MatCardModule, MatButton, NgIf, HttpClientModule, MatProgressSpinner, MatIcon, NgForOf, NgStyle],
  templateUrl: './home.component.html',
  standalone: true,
  styleUrl: './home.component.css'
})
export class HomeComponent {
  selectedFile: File | null = null;
  previewUrl: string | null = null;
  identifiedObject: string | null = null;
  isLoading = false;

  // Waste Category Hints during detection
  loadingHints = ['Detecting Plastic...', 'Detecting Glass...', 'Detecting Paper...', 'Detecting Metal...'];
  currentHintIndex = 0;
  intervalId: any;

  constructor(private wasteService: ApiService) { }

  // File selection event
  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      const reader = new FileReader();
      reader.onload = () => {
        this.previewUrl = reader.result as string;
      };
      reader.readAsDataURL(file);
    }
  }

  // Trigger file input
  triggerFileInput() {
    const fileInput = document.getElementById('file-upload') as HTMLElement;
    fileInput.click();
  }

  // Drag & Drop Handling
  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    const dropZone = event.currentTarget as HTMLElement;
    dropZone.classList.add("dragover");
  }

  async onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();

    const dropZone = event.currentTarget as HTMLElement;
    dropZone.classList.remove("dragover");

    if (event.dataTransfer?.files.length) {
      // Handle normal file drop
      this.onFileSelected({ target: { files: event.dataTransfer.files } });
    } else if (event.dataTransfer?.items.length) {
      for (let item of event.dataTransfer.items) {
        if (item.kind === "string" && item.type === "text/uri-list") {
          item.getAsString(async (imageUrl: string) => {
            console.log("Dragged image URL:", imageUrl);

            // **Check if URL is a direct image file**
            if (!imageUrl.match(/\.(jpeg|jpg|gif|png|webp)$/)) {
              console.error("This is not a direct image URL. Cannot fetch.");
              return;
            }

            const imageFile = await this.urlToFile(imageUrl);
            if (imageFile) {
              this.onFileSelected({ target: { files: [imageFile] } });
            }
          });
        }
      }
    }
  }

  // Identify Image Process (calls the service)
  async onIdentify() {
    if (!this.selectedFile) return;

    this.isLoading = true;
    this.identifiedObject = null;
    this.currentHintIndex = 0;

    // Start hint animations
    this.intervalId = setInterval(() => {
      this.currentHintIndex = (this.currentHintIndex + 1) % this.loadingHints.length;
    }, 1000);

    try {
      // Create a delay promise that waits for at least 3 seconds
      const delay = new Promise((resolve) => setTimeout(resolve, 3000));

      // Fetch result from API
      const apiCall = this.wasteService.identifyWaste(this.selectedFile);

      // Ensure both API call and delay finish
      const result = await Promise.all([apiCall, delay]);

      // API result is the first value from the Promise.all array
      this.identifiedObject = result[0];

    } finally {
      clearInterval(this.intervalId);
      this.isLoading = false;
    }
  }


  async urlToFile(imageUrl: string): Promise<File | null> {
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
      }

      const blob = await response.blob();
      const fileType = blob.type || "image/png";

      return new File([blob], "dragged-image.png", { type: fileType });
    } catch (error) {
      console.error("Error fetching image:", error);
      return null;
    }
  }

}
