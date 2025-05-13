import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Link } from "lucide-react";
import { SongCard } from "@/components/SongCard";

// Temporary mock data for demonstration
const mockSongs = [
  { title: "Song 1", artist: "Artist 1" },
  { title: "Song 2", artist: "Artist 2" },
  { title: "Song 3", artist: "Artist 3" },
];

export default function Index() {
  const [blogContent, setBlogContent] = useState("");
  const [playlistUrl, setPlaylistUrl] = useState("");
  const [songs, setSongs] = useState(mockSongs);

  const handleGeneratePlaylist = () => {
    // This will be connected to your backend later
    console.log("Generating playlist from:", { blogContent, playlistUrl });
  };

  const handleLike = (index: number) => {
    console.log("Liked song:", songs[index]);
  };

  const handleDislike = (index: number) => {
    console.log("Disliked song:", songs[index]);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="container max-w-4xl mx-auto py-12 px-4">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
            Personalized Playlist Generator
          </h1>
          <p className="text-muted-foreground">
            Share your thoughts, and we'll create a playlist that matches your vibe
          </p>
        </div>

        <Card className="p-6 mb-8">
          <div className="space-y-4">
            <div className="relative">
              <Input
                placeholder="Paste your Spotify playlist URL..."
                value={playlistUrl}
                onChange={(e) => setPlaylistUrl(e.target.value)}
                className="pl-9"
              />
              <Link className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
            </div>
            <Textarea
              placeholder="Paste your blog content here..."
              className="min-h-[200px]"
              value={blogContent}
              onChange={(e) => setBlogContent(e.target.value)}
            />
            <Button 
              onClick={handleGeneratePlaylist}
              className="w-full bg-purple-500 hover:bg-purple-600"
            >
              Generate Playlist
            </Button>
          </div>
        </Card>

        {songs.length > 0 && (
          <div>
            <h2 className="text-2xl font-semibold mb-6">Your Personalized Playlist</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {songs.map((song, index) => (
                <SongCard
                  key={index}
                  song={song}
                  onLike={() => handleLike(index)}
                  onDislike={() => handleDislike(index)}
                />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
