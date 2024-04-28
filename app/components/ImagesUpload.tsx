"use client";

import React, { useState, ChangeEvent } from 'react';
type ImageState = File[];

export default () => {
    const [images, setImages] = useState<ImageState>([]);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            const newImages = event.target.files
            setImages(prevImages => [...prevImages, ...Array.from(newImages)]);
        }
    };

    const removeFile = (fileName: string) => {
        setImages(prevImages => prevImages.filter(file => file.name !== fileName));
    };

    return (
        // container mx-auto p-4
        <div className="flex flex-col justify-center items-center ">
            <div>
            <h1 className="m-10 text-7xl">Upload Images</h1>
            {images.map((file, index) => (
                <div key={index} className="flex items-center space-x-2 p-2">
                    <span className="flex-1">{file.name}</span>
                    <button onClick={() => removeFile(file.name)} className="px-2 py-1 bg-red-500 text-white rounded">X</button>
                </div>
            ))}
            </div>
            <input type="file" onChange={handleFileChange} className="
                block text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-300"/>
            <button className="mt-4 px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-300">Done</button>
        </div>
    );
};
