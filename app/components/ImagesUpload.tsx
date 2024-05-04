"use client";

import Link from "next/link";
import React, { useState, ChangeEvent } from 'react';

export default ({ images, setImages }: { images: any[], setImages: any }) => {
    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        console.log('yes')
        if (event.target.files) {
            const newImages = event.target.files
            setImages((prevImages: any) => [...prevImages, ...Array.from(newImages)]);
        }
    };

    const removeFile = (index: number) => {
        setImages(images.slice(0, index).concat(images.slice(index + 1)));
    };

    return (
        <div className="flex flex-col justify-center items-center ">
            <div>
            <h1 className="m-10 text-7xl">Upload Images</h1>
            {images.map((file, index) => (
                <div key={index} className="flex items-center space-x-2 p-2">
                    <span className="flex-1">{file.name}</span>
                    <button onClick={() => removeFile(index)} className="px-2 py-1 bg-red-500 text-white rounded">X</button>
                </div>
            ))}
            </div>
            <input type="file" onChange={handleFileChange} className="
                block text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-300"/>
            <Link className="mt-4 px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-300" href={"/isbn-review"}>Done</Link>
        </div>
    );
};
