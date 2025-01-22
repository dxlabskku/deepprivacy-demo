const useGapText = (createdAt: string): string => {
    const gap = Date.now() - Date.parse(createdAt);
    if (gap >= 604800000) {
        return `${Math.floor(gap / 604800000)} weeks`;
    } else if (gap >= 86400000) {
        return `${Math.floor(gap / 86400000)} days`;
    } else if (gap >= 3600000) {
        return `${Math.floor(gap / 3600000)} hours`;
    } else if (gap >= 60000) {
        return `${Math.floor(gap / 60000)} minutes`;
    } else {
        return "Just now";
    }
};

export default useGapText;
