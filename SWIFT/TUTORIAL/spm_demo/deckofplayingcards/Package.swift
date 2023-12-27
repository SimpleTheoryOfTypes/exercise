// swift-tools-version:5.3

/*
 This source file is part of the Swift.org open source project

 Copyright 2015 – 2021 Apple Inc. and the Swift project authors
 Licensed under Apache License v2.0 with Runtime Library Exception

 See http://swift.org/LICENSE.txt for license information
 See http://swift.org/CONTRIBUTORS.txt for Swift project authors
*/

import PackageDescription

let package = Package(
    name: "DeckOfPlayingCards",
    products: [
        .library(name: "DeckOfPlayingCards", targets: ["DeckOfPlayingCards"]),
    ],
    dependencies: [
        .package(path: "../playingcard"),
    ],
    targets: [
        .target(
            name: "DeckOfPlayingCards",
            dependencies: [
                .product(name: "PlayingCard", package: "playingcard")
            ]),
        .testTarget(
            name: "DeckOfPlayingCardsTests",
            dependencies: [
                .target(name: "DeckOfPlayingCards")
            ]),
    ]
)
