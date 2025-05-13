
import { ThumbsUp, ThumbsDown, Music2 } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

interface SongCardProps {
  song: {
    title: string;
    artist: string;
  };
  onLike: () => void;
  onDislike: () => void;
  className?: string;
}

export function SongCard({ song, onLike, onDislike, className }: SongCardProps) {
  return (
    <div className={cn("bg-card p-4 rounded-lg shadow-lg flex flex-col gap-2", className)}>
      <div className="bg-purple-900/20 rounded-full w-12 h-12 flex items-center justify-center mb-2">
        <Music2 className="text-purple-400 w-6 h-6" />
      </div>
      <h3 className="font-semibold text-lg">{song.title}</h3>
      <p className="text-sm text-muted-foreground">{song.artist}</p>
      <div className="flex gap-2 mt-2">
        <Button variant="outline" size="icon" onClick={onLike}>
          <ThumbsUp className="w-4 h-4" />
        </Button>
        <Button variant="outline" size="icon" onClick={onDislike}>
          <ThumbsDown className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
