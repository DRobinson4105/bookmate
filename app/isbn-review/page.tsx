"use client"

import { useState, useEffect } from "react";
import { useGlobalContext } from "../context/GlobalContext";
import axios from "axios";
import BoundingBoxesComponent from "../components/ImageDisplay";

export default function Home() {
	const { images, setImages, isbns, setIsbns } = useGlobalContext();

    return (
		// <main className="flex min-h-screen flex-col items-center justify-between p-24">
		<main>
			{images.map((file, index) => (
				<div key={index} className="mb-4">
					<div className="flex items-center space-x-2 p-2">
					<span className="flex-1">{file.name}</span>
					</div>
				</div>
				
				))}
				{isbns.map((imageSet, index) => (
					<div>
						{imageSet.map((isbn, idx) => (
							<h1>{isbn[0]}</h1>
						))}
					</div>
				))}
				{images.length > 0 ? (
					<BoundingBoxesComponent image={images[0]} />
				) : (<div />)}
		</main>
    );
  }