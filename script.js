function fillMissingData(data) {
    // Extract timestamps and values from data
    const timestamps1 = data[0];
    const values1 = data[1];
    const timestamps2 = data[2];
    const values2 = data[3];

    // Find missing timestamps in series 1
    const missingTimestamps1 = timestamps2.filter(timestamp => !timestamps1.includes(timestamp));

    // Find missing timestamps in series 2
    const missingTimestamps2 = timestamps1.filter(timestamp => !timestamps2.includes(timestamp));

    // Fill missing timestamps in series 1 using forward fill
    missingTimestamps1.forEach(timestamp => {
        const index = timestamps2.findIndex(ts => ts > timestamp);
        if (index > 0) {
            const interpolatedValue = values2[index - 1];
            timestamps1.push(timestamp);
            values1.push(interpolatedValue);
        }
    });

    // Fill missing timestamps in series 2 using forward fill
    missingTimestamps2.forEach(timestamp => {
        const index = timestamps1.findIndex(ts => ts > timestamp);
        if (index > 0) {
            const interpolatedValue = values1[index - 1];
            timestamps2.push(timestamp);
            values2.push(interpolatedValue);
        }
    });

    // Sort timestamps in both series
    timestamps1.sort();
    timestamps2.sort();

    // Return filled data
    return [timestamps1, values1, timestamps2, values2];
}

// Example usage
const data = [
    [1, 3, 5], // Timestamps for series 1
    [10, 20, 30], // Values for series 1
    [1, 2, 4, 5], // Timestamps for series 2
    [100, 200, 300, 400] // Values for series 2
];

const filledData = fillMissingData(data);
console.log(filledData);